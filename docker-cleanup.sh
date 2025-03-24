#!/bin/bash

# Docker resource cleanup script
# This script helps manage Docker resources and optimize space usage

echo "=============================================="
echo "  Docker Resource Cleanup Utility"
echo "=============================================="
echo ""

# Function to read yes/no input with a default value
read_yes_no() {
    local prompt=$1
    local default=$2
    local input
    
    while true; do
        echo -n "$prompt (y/n) [$default]: "
        read input
        input=${input:-$default}
        case $input in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# Display current Docker disk usage
echo "Current Docker disk usage:"
docker system df
echo ""

# Cleanup options
echo "Cleanup options:"
echo "1. Remove unused containers"
echo "2. Remove unused images"
echo "3. Remove unused volumes"
echo "4. Remove unused networks"
echo "5. Complete system cleanup (containers, images, volumes, networks)"
echo "6. Advanced cleanup (includes builder cache)"
echo "7. Exit"
echo ""

read -p "Select an option (1-7): " option
echo ""

case $option in
    1)
        echo "Removing unused containers..."
        if read_yes_no "Are you sure you want to remove all stopped containers?" "y"; then
            docker container prune -f
            echo "Containers removed successfully."
        else
            echo "Operation cancelled."
        fi
        ;;
    2)
        echo "Removing unused images..."
        if read_yes_no "Are you sure you want to remove all dangling images?" "y"; then
            docker image prune -f
            
            if read_yes_no "Do you also want to remove all unused images (not just dangling ones)?" "n"; then
                docker image prune -af
                echo "All unused images removed successfully."
            else
                echo "Only dangling images removed."
            fi
        else
            echo "Operation cancelled."
        fi
        ;;
    3)
        echo "Removing unused volumes..."
        if read_yes_no "Are you sure you want to remove all unused volumes?" "y"; then
            docker volume prune -f
            echo "Volumes removed successfully."
        else
            echo "Operation cancelled."
        fi
        ;;
    4)
        echo "Removing unused networks..."
        if read_yes_no "Are you sure you want to remove all unused networks?" "y"; then
            docker network prune -f
            echo "Networks removed successfully."
        else
            echo "Operation cancelled."
        fi
        ;;
    5)
        echo "Complete system cleanup..."
        if read_yes_no "Are you sure you want to remove all unused Docker resources?" "y"; then
            docker system prune -f
            
            if read_yes_no "Do you also want to remove all unused volumes?" "n"; then
                docker system prune -f --volumes
                echo "All unused resources including volumes removed successfully."
            else
                echo "All unused resources (except volumes) removed successfully."
            fi
        else
            echo "Operation cancelled."
        fi
        ;;
    6)
        echo "Advanced cleanup..."
        if read_yes_no "Are you sure you want to perform advanced cleanup (including builder cache)?" "y"; then
            docker system prune -af --volumes
            echo "Advanced cleanup completed successfully."
        else
            echo "Operation cancelled."
        fi
        ;;
    7)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "Updated Docker disk usage:"
docker system df
echo ""
echo "=============================================="
echo "  Cleanup process completed!"
echo "=============================================="
